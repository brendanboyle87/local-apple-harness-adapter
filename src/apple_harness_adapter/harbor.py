"""Run native Pi in offline Harbor containers and collect OpenEnv traces."""

import asyncio
import json
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.resources import files

import httpx
from harbor.environments import factory
from harbor.environments.docker import COMPOSE_NO_NETWORK_PATH
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode
from openenv.core.harness.capture.server import create_app
from openenv.harbor.install_fixes import InterceptPi
from openenv.harbor.rollout import run_rollout

from .endpoint import MODEL_NAME, chat_completion
from .trace import Rollout, Turn

PI_VERSION = "1.5.2"
PI_RELAY_URL = "http://127.0.0.1:18765"
CAPTURE_CLIENT = ContextVar("capture_client")


class OfflineDocker(DockerEnvironment):
    """Enforce static Docker isolation even on Desktop kernels without Harbor's egress sidecar."""

    @classmethod
    def _egress_control_kernel_support(cls) -> bool:
        return False

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"disable_internet": True, "stream": False})

    @property
    def _docker_compose_paths(self):
        return [*super()._docker_compose_paths, COMPOSE_NO_NETWORK_PATH]

    def validate_network_policy_support(self, network_policy=None) -> None:
        policy = network_policy or self.network_policy
        if policy.network_mode != NetworkMode.NO_NETWORK or self._is_windows_container:
            raise ValueError(
                "Offline Harbor trials require Linux containers and no-network in every phase"
            )

    async def _apply_network_policy(self, network_policy) -> None:
        self.validate_network_policy_support(network_policy)


@contextmanager
def offline_docker():
    # OpenEnv's runner selects a registered Harbor backend by name. The 0.25
    # Docker registry entry points that name at our static implementation for this run.
    entry = factory._ENVIRONMENT_REGISTRY[EnvironmentType.DOCKER]
    factory._ENVIRONMENT_REGISTRY[EnvironmentType.DOCKER] = entry._replace(
        module=__name__, class_name="OfflineDocker"
    )
    try:
        yield
    finally:
        factory._ENVIRONMENT_REGISTRY[EnvironmentType.DOCKER] = entry


class OfflinePi(InterceptPi):
    async def install(self, environment) -> None:
        result = await self.exec_as_agent(environment, ". ~/.nvm/nvm.sh; pi --version")
        if result.return_code != 0 or (result.stdout or "").strip() != PI_VERSION:
            raise RuntimeError(f"The task image must preinstall Pi {PI_VERSION}")

    async def run(self, instruction, environment, context) -> None:
        if not isinstance(environment, DockerEnvironment):
            raise RuntimeError("Offline Pi requires Harbor's Docker backend")
        container = await environment._run_docker_compose_command(["ps", "--quiet", "main"])
        container_id = container.stdout.strip()
        if not container_id or any(char not in "0123456789abcdef" for char in container_id):
            raise RuntimeError("Could not identify the Harbor container")
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "inspect",
            "--format",
            "{{.HostConfig.NetworkMode}}",
            container_id,
            stdout=asyncio.subprocess.PIPE,
        )
        network, _ = await inspect.communicate()
        if inspect.returncode or network.strip() != b"none":
            raise RuntimeError("Pi requires a Docker container with network mode `none`")

        relay = files("apple_harness_adapter").joinpath("relay.js").read_text(encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--user",
            "root",
            container_id,
            "node",
            "-e",
            relay,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=8 * 2**20,
        )
        try:
            if await asyncio.wait_for(process.stdout.readline(), timeout=10) != b"ready\n":
                raise RuntimeError("Pi's loopback relay did not start")
            bridge = asyncio.create_task(self._forward(process))
            agent = asyncio.create_task(super().run(instruction, environment, context))
            try:
                done, _ = await asyncio.wait((bridge, agent), return_when=asyncio.FIRST_COMPLETED)
                if bridge in done:
                    bridge.result()
                    raise RuntimeError("Pi's loopback relay exited during the rollout")
                agent.result()
            finally:
                for task in (bridge, agent):
                    task.cancel()
                await asyncio.gather(bridge, agent, return_exceptions=True)
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()

    @staticmethod
    async def _forward(process) -> None:
        client = CAPTURE_CLIENT.get()
        while line := await process.stdout.readline():
            request = json.loads(line)
            response = await client.post(
                "/v1/chat/completions",
                content=request["body"],
                headers={
                    "Authorization": request.get("authorization", ""),
                    "Content-Type": "application/json",
                },
            )
            frame = {
                "id": request["id"],
                "status": response.status_code,
                "contentType": response.headers.get("Content-Type", "application/json"),
                "body": response.text,
            }
            process.stdin.write(json.dumps(frame).encode() + b"\n")
            await process.stdin.drain()


def training_rollout(result) -> Rollout:
    if not result.ok or result.reward is None or not result.trainable:
        raise RuntimeError(
            f"Harbor rollout {result.trial_name} failed: {result.error or result.findings}"
        )
    turns = []
    for row in result.turns:
        if row.discarded or not row.trainable or row.role != "agent":
            continue
        prompt = tuple(row.prompt_token_ids)
        completion = tuple(row.completion_token_ids)
        if row.loss_mask != [0] * len(prompt) + [1] * len(completion):
            raise RuntimeError(f"Unexpected partial completion mask in {result.trial_name}")
        turns.append(
            Turn(prompt, completion, tuple(row.per_token_logps), row.sampling_params["temperature"])
        )
    if (
        not turns
        or sum(len(turn.completion_token_ids) for turn in turns) != result.n_trainable_tokens
    ):
        raise RuntimeError(f"Incomplete trainable token accounting in {result.trial_name}")
    return Rollout(
        result.task_name,
        tuple(turns),
        result.reward,
        sum(
            len(row.tool_calls) for row in result.turns if not row.discarded and row.role == "agent"
        ),
    )


async def collect_harbor(policy, args, trials_dir, save_attempt, task) -> list[Rollout]:
    def inference_request(request):
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        try:
            payload = chat_completion(
                policy,
                json.loads(request.content),
                max_context=args.max_context,
                max_output=args.max_output,
                effort=args.effort,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return httpx.Response(400, json={"error": {"message": str(exc)}})
        return httpx.Response(200, json=payload)

    with offline_docker():
        engine = "http://mlx.local/v1"
        capture = create_app(
            llm_url=engine,
            model=MODEL_NAME,
            capture_level="tokens",
            max_output_tokens=args.max_output,
            admin_key=secrets.token_urlsafe(32),
        )
        # Keep MLX operations on its owning thread. OpenEnv still prepares,
        # normalizes and validates every inference request and response.
        capture.state.inference._client = httpx.AsyncClient(
            transport=httpx.MockTransport(inference_request),
            base_url=engine,
        )
        transport = httpx.ASGITransport(app=capture)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://capture.local"
        ) as client:
            reset = CAPTURE_CLIENT.set(client)
            try:
                rollouts = []
                for attempt in range(args.attempts):
                    result = await run_rollout(
                        task_dir=task,
                        harness="apple_harness_adapter.harbor:OfflinePi",
                        sandbox="docker",
                        registry=capture.state.registry,
                        intercept_url=PI_RELAY_URL,
                        model=MODEL_NAME,
                        trials_dir=trials_dir,
                        reward_key=args.reward_key,
                        purpose="train",
                        inference=capture.state.inference,
                        sampling={"temperature": args.temperature},
                        agent_step_limit=args.max_turns,
                    )
                    save_attempt(attempt, result.model_dump(mode="json"))
                    rollouts.append(training_rollout(result))
                return rollouts
            finally:
                CAPTURE_CLIENT.reset(reset)
                await capture.state.inference.aclose()
