package com.auth.oauth2.entity;


public class LogSt {

  private long logId;
  private long userId;
  private String copDetail;
  private java.sql.Timestamp copTime;
  private int copType;


  public long getLogId() {
    return logId;
  }

  public void setLogId(long logId) {
    this.logId = logId;
  }


  public long getUserId() {
    return userId;
  }

  public void setUserId(long userId) {
    this.userId = userId;
  }


  public String getCopDetail() {
    return copDetail;
  }

  public void setCopDetail(String copDetail) {
    this.copDetail = copDetail;
  }


  public java.sql.Timestamp getCopTime() {
    return copTime;
  }

  public void setCopTime(java.sql.Timestamp copTime) {
    this.copTime = copTime;
  }

  public int getCopType() {
    return copType;
  }

  public void setCopType(int copType) {
    this.copType = copType;
  }
}
