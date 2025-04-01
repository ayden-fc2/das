FROM openjdk:8-jdk-alpine

WORKDIR /app

COPY target/auth-service-*.jar /app/auth-service.jar

EXPOSE 2074

ENTRYPOINT ["sh", "-c", "java -jar /app/auth-service.jar"]