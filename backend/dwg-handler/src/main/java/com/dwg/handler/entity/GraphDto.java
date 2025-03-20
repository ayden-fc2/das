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
    private JSONObject stream;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public JSONObject getPosition() {
        return position;
    }

    public void setPosition(JSONObject position) {
        this.position = position;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public BlockSt getNode() {
        return node;
    }

    public void setNode(BlockSt node) {
        this.node = node;
    }

    public JSONObject getBox() {
        return box;
    }

    public void setBox(JSONObject box) {
        this.box = box;
    }

    public JSONObject getStream() {
        return stream;
    }

    public void setStream(JSONObject stream) {
        this.stream = stream;
    }
}

