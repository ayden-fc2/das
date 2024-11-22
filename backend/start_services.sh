#!/bin/bash

# TODO select your own directory
BASE_DIR=$(pwd)

services=("example2060" "service-discovery" "api-gateway" "auth-service")

for service in "${services[@]}"; do
    echo "Attempting to navigate to $BASE_DIR/$service"
    alacritty -t "$service - Terminal" -e zsh -c "cd $BASE_DIR/$service && mvn spring-boot:run" &
done

# http://localohost:2072 eureka
# http://localohost:2073 zuul
# http://localohost:2074 auth
