package com.dwg.handler.entity;

import lombok.Data;

@Data
public class BlockSt {

  private long blockId;
  private String blockName;
  private long blockCount;
  private String svg;
  private long handle0;
  private long handle1;
  private long dwgId;


  public long getBlockId() {
    return blockId;
  }

  public void setBlockId(long blockId) {
    this.blockId = blockId;
  }


  public String getBlockName() {
    return blockName;
  }

  public void setBlockName(String blockName) {
    this.blockName = blockName;
  }


  public long getBlockCount() {
    return blockCount;
  }

  public void setBlockCount(long blockCount) {
    this.blockCount = blockCount;
  }


  public String getSvg() {
    return svg;
  }

  public void setSvg(String svg) {
    this.svg = svg;
  }


  public long getHandle0() {
    return handle0;
  }

  public void setHandle0(long handle0) {
    this.handle0 = handle0;
  }


  public long getHandle1() {
    return handle1;
  }

  public void setHandle1(long handle1) {
    this.handle1 = handle1;
  }


  public long getDwgId() {
    return dwgId;
  }

  public void setDwgId(long dwgId) {
    this.dwgId = dwgId;
  }

}
