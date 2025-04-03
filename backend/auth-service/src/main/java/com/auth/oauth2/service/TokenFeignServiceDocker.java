package com.auth.oauth2.service;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.Map;

@FeignClient(name = "jwt-token-api-docker", url = "http://auth-service:2074")
public interface TokenFeignServiceDocker {
    @PostMapping("/oauth/token")
    String getToken(@RequestParam Map<String, String> request);
}
