package com.file.manage;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;

@SpringBootApplication(scanBasePackages = {"com.example.common", "com.file.manage"})
@EnableDiscoveryClient
@EnableResourceServer
public class FileManageApplication {

    public static void main(String[] args) {
        SpringApplication.run(FileManageApplication.class, args);
    }

}
