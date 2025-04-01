FROM openjdk:8-jdk-alpine

WORKDIR /app

COPY target/dwg-handler-*.jar /app/dwg-handler.jar

EXPOSE 2075

ENTRYPOINT ["sh", "-c", "java -jar /app/dwg-handler.jar"]