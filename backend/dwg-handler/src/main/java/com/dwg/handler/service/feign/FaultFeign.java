package com.dwg.handler.service.feign;

import com.alibaba.fastjson.JSONObject;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.Map;

@FeignClient(name = "fault-feign", url = "http://localhost:2081")
public interface FaultFeign {
    @PostMapping("/predict")
    JSONObject getResult(@RequestBody JSONObject request);
}
