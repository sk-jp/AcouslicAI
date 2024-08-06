FROM python:3.11-slim
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# FROM --platform=linux/amd64 pytorch/pytorch
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user calc_circum.py /opt/app/
COPY --chown=user:user ellipse_circumference.py /opt/app/
COPY --chown=user:user fix_model_state_dict.py /opt/app/
COPY --chown=user:user unet_multitask.py /opt/app/
COPY --chown=user:user get_transform.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user read_yaml.py /opt/app/

COPY --chown=user:user unet_multitask_3slices_test.yaml /opt/app/
COPY --chown=user:user unet_multitask_3slices-LR2e-05-base_W1_2_005_FL-epoch=49-valid_loss=0.74.ckpt /opt/app/

ENTRYPOINT ["python", "inference.py"]
