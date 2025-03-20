package com.dwg.handler.entity;

import lombok.Data;

@Data
public class KeyPipeSt implements Cloneable {
    private long keyPipeId;
    private double startX;
    private double startY;
    private double endX;
    private double endY;
    private long dwgId;

    private long startKeyHandle0 = -1;
    private long startKeyHandle1 = -1;
    private long endKeyHandle0 = -1;
    private long endKeyHandle1 = -1;

    private String vStartUUID;
    private String vEndUUID;

    public long getKeyPipeId() {
        return keyPipeId;
    }

    public void setKeyPipeId(long keyPipeId) {
        this.keyPipeId = keyPipeId;
    }

    public double getStartX() {
        return startX;
    }

    public void setStartX(double startX) {
        this.startX = startX;
    }

    public double getStartY() {
        return startY;
    }

    public void setStartY(double startY) {
        this.startY = startY;
    }

    public double getEndX() {
        return endX;
    }

    public void setEndX(double endX) {
        this.endX = endX;
    }

    public double getEndY() {
        return endY;
    }

    public void setEndY(double endY) {
        this.endY = endY;
    }

    public long getDwgId() {
        return dwgId;
    }

    public void setDwgId(long dwgId) {
        this.dwgId = dwgId;
    }

    public long getStartKeyHandle0() {
        return startKeyHandle0;
    }

    public void setStartKeyHandle0(long startKeyHandle0) {
        this.startKeyHandle0 = startKeyHandle0;
    }

    public long getStartKeyHandle1() {
        return startKeyHandle1;
    }

    public void setStartKeyHandle1(long startKeyHandle1) {
        this.startKeyHandle1 = startKeyHandle1;
    }

    public long getEndKeyHandle0() {
        return endKeyHandle0;
    }

    public void setEndKeyHandle0(long endKeyHandle0) {
        this.endKeyHandle0 = endKeyHandle0;
    }

    public long getEndKeyHandle1() {
        return endKeyHandle1;
    }

    public void setEndKeyHandle1(long endKeyHandle1) {
        this.endKeyHandle1 = endKeyHandle1;
    }

    public String getVStartUUID() {
        return vStartUUID;
    }

    public void setVStartUUID(String vStartUUID) {
        this.vStartUUID = vStartUUID;
    }

    public String getVEndUUID() {
        return vEndUUID;
    }

    public void setVEndUUID(String vEndUUID) {
        this.vEndUUID = vEndUUID;
    }

    @Override
    public KeyPipeSt clone() {
        try {
            return (KeyPipeSt) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }

}
