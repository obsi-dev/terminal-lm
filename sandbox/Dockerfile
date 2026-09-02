# sandbox/Dockerfile
FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends coreutils findutils grep sed gawk tar gzip bzip2 xz-utils curl wget rsync git openssh-client
RUN rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser
