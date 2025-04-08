package com.dwg.handler;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;

@SpringBootApplication(scanBasePackages = {"com.example.common", "com.dwg.handler"})
@EnableDiscoveryClient
@EnableResourceServer
@EnableFeignClients(basePackages = {"com.dwg.handler.service.feign"})
public class DwgHandlerApplication {

    public static void main(String[] args) {
        SpringApplication.run(DwgHandlerApplication.class, args);
    }

}
