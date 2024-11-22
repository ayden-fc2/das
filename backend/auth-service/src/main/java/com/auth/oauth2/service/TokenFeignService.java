package com.auth.oauth2.service;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.Map;

@FeignClient(name = "jwt-token-api", url = "http://localhost:2074")
public interface TokenFeignService {
    @PostMapping("/oauth/token")
    String getToken(@RequestParam Map<String, String> request);
}
