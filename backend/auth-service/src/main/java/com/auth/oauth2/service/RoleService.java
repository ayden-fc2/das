package com.auth.oauth2.service;

import com.alibaba.fastjson.JSONObject;

public interface RoleService {
    JSONObject getUserBaseInfo(int userId);
}
