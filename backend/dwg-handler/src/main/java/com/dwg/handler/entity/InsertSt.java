package com.dwg.handler.entity;

import lombok.Data;

@Data
public class InsertSt {

  private long insertId;
  private long blockHandle0;
  private long blockHandle1;
  private long insertHandle0;
  private long insertHandle1;
  private double boxWidth;
  private double boxHeight;
  private double centerPtX;
  private double centerPtY;
  private long dwgId;


  public long getInsertId() {
    return insertId;
  }

  public void setInsertId(long insertId) {
    this.insertId = insertId;
  }


  public long getBlockHandle0() {
    return blockHandle0;
  }

  public void setBlockHandle0(long blockHandle0) {
    this.blockHandle0 = blockHandle0;
  }


  public long getBlockHandle1() {
    return blockHandle1;
  }

  public void setBlockHandle1(long blockHandle1) {
    this.blockHandle1 = blockHandle1;
  }


  public long getInsertHandle0() {
    return insertHandle0;
  }

  public void setInsertHandle0(long insertHandle0) {
    this.insertHandle0 = insertHandle0;
  }


  public long getInsertHandle1() {
    return insertHandle1;
  }

  public void setInsertHandle1(long insertHandle1) {
    this.insertHandle1 = insertHandle1;
  }


  public double getBoxWidth() {
    return boxWidth;
  }

  public void setBoxWidth(double boxWidth) {
    this.boxWidth = boxWidth;
  }


  public double getBoxHeight() {
    return boxHeight;
  }

  public void setBoxHeight(double boxHeight) {
    this.boxHeight = boxHeight;
  }


  public double getCenterPtX() {
    return centerPtX;
  }

  public void setCenterPtX(double centerPtX) {
    this.centerPtX = centerPtX;
  }


  public double getCenterPtY() {
    return centerPtY;
  }

  public void setCenterPtY(double centerPtY) {
    this.centerPtY = centerPtY;
  }

  public long getDwgId() {
    return dwgId;
  }

  public void setDwgId(long dwgId) {
    this.dwgId = dwgId;
  }

}
