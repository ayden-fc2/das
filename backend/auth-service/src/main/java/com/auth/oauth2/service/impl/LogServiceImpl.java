package com.auth.oauth2.service.impl;

import com.auth.oauth2.entity.LogSt;
import com.auth.oauth2.mapper.LogMapper;
import com.auth.oauth2.service.LogService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

@Service
public class LogServiceImpl implements LogService {

    @Autowired
    LogMapper logMapper;

    @Override
    public Boolean addLog(int userId,String copDetail, int copType) {
        return logMapper.addLog(userId, copDetail, copType);
    }

    @Override
    public List<LogSt> queryLog(int userId, String startDate, String endDate) {
        return logMapper.queryLog(userId, startDate, endDate);
    }
}
