package com.dwg.handler.entity;

import com.alibaba.fastjson.JSONObject;
import lombok.Data;

@Data
public class GraphDto {
    private String id;
    private String label;
    private String type;
    private JSONObject position;
    private String source;
    private String target;
    private BlockSt node;
    private JSONObject box;
}

