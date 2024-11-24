package com.example.common.exception;


import com.example.common.dto.ResponseBean;


public class MyException extends RuntimeException{
    /**
     * 提示信息
     */
    private String message;

    public MyException() {
        this.message = "未知错误";
    }
    public MyException(String message) {
        this.message = message;
    }


    public ResponseBean getFailResponse() {
        if (null != getMessage() && !getMessage().isEmpty()) {
            return ResponseBean.fail(getMessage());
        }
        return ResponseBean.fail(message);
    }
}
