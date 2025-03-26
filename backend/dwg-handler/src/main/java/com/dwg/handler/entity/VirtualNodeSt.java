package com.dwg.handler.entity;

import lombok.Data;

@Data
public class VirtualNodeSt {
    private long vNodeId;
    private double x;
    private double y;
    private String uuid;
    private long dwgId;
    private long finished;

    public long getVNodeId() {
        return vNodeId;
    }

    public void setVNodeId(long vNodeId) {
        this.vNodeId = vNodeId;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public long getDwgId() {
        return dwgId;
    }

    public void setDwgId(long dwgId) {
        this.dwgId = dwgId;
    }

    public long getFinished() {
        return finished;
    }

    public void setFinished(long finished) {
        this.finished = finished;
    }
}
