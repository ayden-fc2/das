package com.auth.oauth2.config;

import com.example.common.dto.ResponseBean;
import com.example.common.exception.MyException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.nio.file.AccessDeniedException;

@RestControllerAdvice
@Slf4j
public class ExceptionControllerAdvice {
    @ExceptionHandler(MyException.class)
    public ResponseBean handleServiceException(MyException e) {
        return e.getFailResponse();
    }

    @ExceptionHandler(Exception.class)
    public ResponseBean handleException(Exception e) {
        if (e instanceof MissingServletRequestParameterException) {
            return ResponseBean.fail("缺少参数"+e.getMessage());
        } else if (e instanceof AccessDeniedException) {
            return ResponseBean.fail("参数不足");
        }else {
            // log.error("拦截异常:", e);
            return ResponseBean.fail(e.getMessage());
        }
    }
}
