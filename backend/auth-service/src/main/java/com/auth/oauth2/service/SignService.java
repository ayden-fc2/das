package com.auth.oauth2.service;

import com.example.common.dto.ResponseBean;

public interface SignService {
    ResponseBean signInCheck(String phoneNum, String password);

    ResponseBean getPhoneCode(String phoneNum, int mode);

    ResponseBean signUp(String phoneNum, int mode, String code, String password, String name);

    ResponseBean resetPassword(String phoneNum, int mode, String code, String newPassword);

    ResponseBean getUserId(String token);

    ResponseBean getUserInfo(String token);
}
