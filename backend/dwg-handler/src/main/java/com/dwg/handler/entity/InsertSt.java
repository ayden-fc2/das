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
  private String upstream;
  private String downstream;

}
