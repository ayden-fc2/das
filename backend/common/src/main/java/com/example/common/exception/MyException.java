package com.example.common.exception;


import com.example.common.dto.ResponseBean;

/**
 * @ClassName MyException
 * @Description 自定义异常类
 * @Author ShenHaoran
 * @Version : v1.0
 * @CreateTime : 2023/12/22 18:05
 */
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
