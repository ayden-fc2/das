package com.example.common.enums;

import lombok.Getter;

@Getter
public enum UserType {
    SuperManager("SUPER_MANAGER_TYPE","超级管理员"),
    Manager("MANAGER_TYPE","管理员"),
    Controller("CONTROLLER_TYPE","控制者"),
    Observer("OBSERVER_TYPE","观察者");

    private final String type;
    private final String message;

    UserType(String type, String message) {
        this.type = type;
        this.message = message;
    }

}
