FROM openjdk:8-jdk-alpine

WORKDIR /app

COPY target/service-discovery-*.jar /app/service-discovery.jar

EXPOSE 2072

ENTRYPOINT ["sh", "-c", "java -jar /app/service-discovery.jar --eureka.instance.hostname=${HOSTNAME:-eureka}"]