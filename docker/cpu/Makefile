include .env
AWS_VERSION=v2.4
AWS_NAME=kz_km_predict

DOCKERHUB_VERSION=v2.4


aws-build:
	docker-compose build

aws-login:
	aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin $(AWS_REPOSITORY)

aws-push:
	docker tag docker_km_predict:latest $(AWS_REPOSITORY)/$(AWS_NAME):$(AWS_VERSION)
	docker push $(AWS_REPOSITORY)/$(AWS_NAME):$(AWS_VERSION)


dockerhub-build:
	docker build -t kappazeta/kappamask:$(DOCKERHUB_VERSION) km_predict

dockerhub-login:
	docker login -u kappazeta

dockerhub-push:
	docker push kappazeta/kappamask:$(DOCKERHUB_VERSION)
