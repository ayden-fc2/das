package com.auth.oauth2.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.mapper.UserMapper;
import com.auth.oauth2.service.RoleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class RoleServiceImpl implements RoleService {
    @Autowired
    UserMapper um;

    @Override
    public JSONObject getUserBaseInfo(int userId) {

        return um.selectBasicInfoByUserId(userId);
    }
}
