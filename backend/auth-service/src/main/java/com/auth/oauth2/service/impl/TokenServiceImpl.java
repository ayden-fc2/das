package com.auth.oauth2.service.impl;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.service.TokenService;
import org.springframework.security.jwt.Jwt;
import org.springframework.security.jwt.JwtHelper;
import org.springframework.stereotype.Service;

@Service
public class TokenServiceImpl implements TokenService {
    public int tokenToUserId(String token) {
        // 去掉 "Bearer " 前缀
        if (token.startsWith("Bearer ")) {
            token = token.substring(7);
        }

        Jwt jwt = JwtHelper.decode(token);
        String jsonString = jwt.getClaims();

        // 使用 FastJSON 解析 JSON 字符串
        JSONObject claimsJson = JSONObject.parseObject(jsonString);

        // 获取 "userId"
        return (Integer.parseInt(claimsJson.getString("userId")));
    }

    public JSONArray tokenToRoles(String token) {
        // 去掉 "Bearer " 前缀
        if (token.startsWith("Bearer ")) {
            token = token.substring(7);
        }

        Jwt jwt = JwtHelper.decode(token);
        String jsonString = jwt.getClaims();

        // 使用 FastJSON 解析 JSON 字符串
        JSONObject claimsJson = JSONObject.parseObject(jsonString);

        JSONArray roles = claimsJson.getJSONArray("authorities");
        // 返回 claimsJson
        return roles;
    }
}
