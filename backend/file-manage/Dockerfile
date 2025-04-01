FROM openjdk:8-jdk-alpine

WORKDIR /app

COPY target/file-manage-*.jar /app/file-manage.jar

EXPOSE 2076

ENTRYPOINT ["sh", "-c", "java -jar /app/file-manage.jar"]