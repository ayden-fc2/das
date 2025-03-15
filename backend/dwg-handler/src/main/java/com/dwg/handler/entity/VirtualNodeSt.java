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
}
