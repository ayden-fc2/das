package com.auth.oauth2.entity;

import lombok.Data;

@Data
public class AccountSt {

  private long accountId;
  private String phoneNum;
  private String passwordDetail;
  private long accountPer;
  private String nickName;
  private String avatar;

  // Getter and Setter for accountId
  public long getAccountId() {
    return accountId;
  }

  public void setAccountId(long accountId) {
    this.accountId = accountId;
  }

  // Getter and Setter for phoneNum
  public String getPhoneNum() {
    return phoneNum;
  }

  public void setPhoneNum(String phoneNum) {
    this.phoneNum = phoneNum;
  }

  // Getter and Setter for passwordDetail
  public String getPasswordDetail() {
    return passwordDetail;
  }

  public void setPasswordDetail(String passwordDetail) {
    this.passwordDetail = passwordDetail;
  }

  // Getter and Setter for accountPer
  public long getAccountPer() {
    return accountPer;
  }

  public void setAccountPer(long accountPer) {
    this.accountPer = accountPer;
  }

  // Getter and Setter for nickName
  public String getNickName() {
    return nickName;
  }

  public void setNickName(String nickName) {
    this.nickName = nickName;
  }

  // Getter and Setter for empty
  public String getAvatar() {
    return avatar;
  }

  public void setAvatar(String avatar) {
    this.avatar = avatar;
  }
}

