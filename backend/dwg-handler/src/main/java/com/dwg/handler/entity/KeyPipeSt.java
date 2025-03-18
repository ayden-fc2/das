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

    @Override
    public KeyPipeSt clone() {
        try {
            return (KeyPipeSt) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }
}
