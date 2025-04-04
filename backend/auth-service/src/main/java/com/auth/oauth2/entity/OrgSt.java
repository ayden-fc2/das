package com.auth.oauth2.entity;


public class OrgSt {

  private long orgId;
  private String orgName;
  private java.sql.Timestamp createdTime;
  private long createrId;
  private String orgDesc;
  private String orgCode;


  public long getOrgId() {
    return orgId;
  }

  public void setOrgId(long orgId) {
    this.orgId = orgId;
  }


  public String getOrgName() {
    return orgName;
  }

  public void setOrgName(String orgName) {
    this.orgName = orgName;
  }


  public java.sql.Timestamp getCreatedTime() {
    return createdTime;
  }

  public void setCreatedTime(java.sql.Timestamp createdTime) {
    this.createdTime = createdTime;
  }


  public long getCreaterId() {
    return createrId;
  }

  public void setCreaterId(long createrId) {
    this.createrId = createrId;
  }


  public String getOrgDesc() {
    return orgDesc;
  }

  public void setOrgDesc(String orgDesc) {
    this.orgDesc = orgDesc;
  }


  public String getOrgCode() {
    return orgCode;
  }

  public void setOrgCode(String orgCode) {
    this.orgCode = orgCode;
  }

}
