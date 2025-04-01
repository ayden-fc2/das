FROM openjdk:8-jdk-alpine

WORKDIR /app

COPY target/api-gateway-*.jar /app/api-gateway.jar

EXPOSE 2073

ENTRYPOINT ["sh", "-c", "java -jar /app/api-gateway.jar"]