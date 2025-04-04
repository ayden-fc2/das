package com.auth.oauth2.service;

import com.auth.oauth2.entity.LogSt;

import java.util.List;

public interface LogService {
    Boolean addLog(int userId, String copDetail, int copType);

    List<LogSt> queryLog(int userId, String startDate, String endDate);
}
