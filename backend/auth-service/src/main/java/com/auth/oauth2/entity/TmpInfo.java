package com.auth.oauth2.entity;

import lombok.Data;

@Data
public class TmpInfo {

    private String accountNumber;
    private String verificationCode;
    private String mode;

    // Getter and Setter for phoneNumber
    public String getAccountNumber() {
        return accountNumber;
    }

    public void setAccountNumber(String phoneNumber) {
        this.accountNumber = phoneNumber;
    }

    // Getter and Setter for verificationCode
    public String getVerificationCode() {
        return verificationCode;
    }

    public void setVerificationCode(String verificationCode) {
        this.verificationCode = verificationCode;
    }

    // Getter and Setter for mode
    public String getMode() {
        return mode;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }
}

