package com.example.common.dto;

import com.alibaba.fastjson.annotation.JSONField;

public class ResponseBean<T> {
    /**
     * 实现功能:是否成功
     */
    @JSONField(name = "success")
    private Integer success;

    /**
     * 实现功能：返回信息
     */
    @JSONField(name = "message")
    protected String message;

    /**
     * 返回数据结构
     */
    @JSONField(name = "data", ordinal = 1)
    protected T data;

    // 全参构造方法（手动定义）
    public ResponseBean(Integer success, String message, T data) {
        this.success = success;
        this.message = message;
        this.data = data;
    }

    // 无参构造方法
    public ResponseBean() {}

    // Getter 和 Setter
    public Integer getSuccess() {
        return success;
    }

    public void setSuccess(Integer success) {
        this.success = success;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    /**
     * 实现功能：常用成功的返回，带额外数据
     *
     * @param data 当前数据
     * @return 请求成功ResponseBean
     */
    public static <TData> ResponseBean<TData> success(TData data) {
        return new ResponseBean<>(1, null, data);
    }

    /**
     * 实现功能：常用成功的返回，不带额外数据
     *
     * @return 请求成功ResponseBean
     */
    public static <TData> ResponseBean<TData> success() {
        return success(null);
    }

    /**
     * 实现功能:常用失败的运回
     *
     * @param message 出错信息
     * @return 请求失败ResponseBean
     */
    public static ResponseBean<Void> fail(String message) {
        return new ResponseBean<>(0, message, null);
    }
}
