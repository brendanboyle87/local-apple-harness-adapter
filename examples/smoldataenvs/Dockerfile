ARG HARNESS_IMAGE=local-apple-harness-pi:fixture
FROM ${HARNESS_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/smoldatavenv
ENV PATH="/opt/smoldatavenv/bin:${PATH}"
RUN pip install --no-cache-dir 'numpy==2.2.6' 'pandas==2.3.3' 'scipy==1.15.3' \
    'scikit-learn==1.7.2' 'openpyxl==3.1.5' 'math-verify==0.8.0' \
    && install -d -o node -g node /workdir \
    && install -d /home/user/input
WORKDIR /workdir
