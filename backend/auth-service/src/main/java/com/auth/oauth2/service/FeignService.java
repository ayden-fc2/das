package com.auth.oauth2.service;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.Map;

@FeignClient(name = "third-party-api", url = "https://gyytz.market.alicloudapi.com")
public interface FeignService {
    @PostMapping("/sms/smsSend")
    String sendCode(@RequestHeader("Authorization") String authorization,
                    @RequestParam Map<String, String> request);

}

