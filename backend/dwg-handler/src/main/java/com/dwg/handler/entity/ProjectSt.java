package com.dwg.handler.entity;


public class ProjectSt {

  private long projectKey;
  private java.sql.Timestamp createdTime;
  private String title;
  private String desc;
  private String parentKey;
  private String childrenKey;
  private long uploadId;
  private long orgId;
  private long createrId;


  public long getProjectKey() {
    return projectKey;
  }

  public void setProjectKey(long projectKey) {
    this.projectKey = projectKey;
  }


  public java.sql.Timestamp getCreatedTime() {
    return createdTime;
  }

  public void setCreatedTime(java.sql.Timestamp createdTime) {
    this.createdTime = createdTime;
  }


  public String getTitle() {
    return title;
  }

  public void setTitle(String title) {
    this.title = title;
  }


  public String getDesc() {
    return desc;
  }

  public void setDesc(String desc) {
    this.desc = desc;
  }


  public String getParentKey() {
    return parentKey;
  }

  public void setParentKey(String parentKey) {
    this.parentKey = parentKey;
  }


  public String getChildrenKey() {
    return childrenKey;
  }

  public void setChildrenKey(String childrenKey) {
    this.childrenKey = childrenKey;
  }


  public long getUploadId() {
    return uploadId;
  }

  public void setUploadId(long uploadId) {
    this.uploadId = uploadId;
  }


  public long getOrgId() {
    return orgId;
  }

  public void setOrgId(long orgId) {
    this.orgId = orgId;
  }


  public long getCreaterId() {
    return createrId;
  }

  public void setCreaterId(long createrId) {
    this.createrId = createrId;
  }

}
