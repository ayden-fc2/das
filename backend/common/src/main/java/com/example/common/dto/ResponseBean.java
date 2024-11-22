package com.example.common.dto;

import com.alibaba.fastjson.annotation.JSONField;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @ClassName ResponseBean
 * @Description 请求Json返回值
 * @Author ShenHaoran
 * @Version : v1.0
 * @CreateTime : 2023/12/22 17:49
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
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
    public static ResponseBean fail(String message) {
        return new ResponseBean(0, message, null);
    }
}
