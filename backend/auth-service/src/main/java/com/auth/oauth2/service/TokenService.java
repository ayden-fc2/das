package com.auth.oauth2.service;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

public interface TokenService {
    int tokenToUserId(String token);

    JSONObject tokenToRoles(String token);
}

