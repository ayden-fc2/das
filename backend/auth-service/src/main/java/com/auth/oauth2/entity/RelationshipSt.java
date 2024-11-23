package com.auth.oauth2.entity;

import lombok.Data;

@Data
public class RelationshipSt {

  private long relationshipId;
  private long accountId;
  private long authorityId;

  // Getter and Setter for relationshipId
  public long getRelationshipId() {
    return relationshipId;
  }

  public void setRelationshipId(long relationshipId) {
    this.relationshipId = relationshipId;
  }

  // Getter and Setter for accountId
  public long getAccountId() {
    return accountId;
  }

  public void setAccountId(long accountId) {
    this.accountId = accountId;
  }

  // Getter and Setter for authorityId
  public long getAuthorityId() {
    return authorityId;
  }

  public void setAuthorityId(long authorityId) {
    this.authorityId = authorityId;
  }
}

