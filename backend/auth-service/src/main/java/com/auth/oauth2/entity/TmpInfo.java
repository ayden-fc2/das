package com.auth.oauth2.entity;

import lombok.Data;

@Data
public class TmpInfo {
    private String phoneNumber;
    private String verificationCode;
    private String mode;
}
