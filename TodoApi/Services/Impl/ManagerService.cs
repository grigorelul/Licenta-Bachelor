using Models;
using Repositories;

namespace Services;

public class ManagerService : IManagerService
{

    private readonly IAttendanceRepository _attendanceRepository;
    private readonly IManagerRepository _managerRepository;
    
    public ManagerService(IUserRepository userRepository, IAttendanceRepository attendanceRepository, IManagerRepository managerRepository)
    {
        _attendanceRepository = attendanceRepository;
        _managerRepository = managerRepository;
    }
    
    public async Task<ManagerDto> GetManagerAsync(Guid id)
    {
        var manager = await _managerRepository.GetManagerAsync(id);
        return ManagerDto.FromManagerToManagerDto(manager);
    }
    
    public async Task<ManagerDto> CreateManagerAsync(ManagerDto managerDto)
    {
        var manager = Manager.FromManagerDtoToManager(managerDto);
        manager = await _managerRepository.CreateManagerAsync(manager);
        return ManagerDto.FromManagerToManagerDto(manager);
    }
    
    public async Task<ManagerDto> UpdateManagerAsync(Guid id, ManagerDto managerDto)
    {
        var manager = await _managerRepository.GetManagerAsync(id);
        if (manager == null)
        {
            throw new Exception("Manager not found");
        }
        manager = Manager.FromManagerDtoToManager(managerDto);
        manager.Id = id;
        manager = await _managerRepository.UpdateManagerAsync(manager);
        return ManagerDto.FromManagerToManagerDto(manager);
    }
    
    public async Task DeleteManagerAsync(Guid id)
    {
        var manager = await _managerRepository.GetManagerAsync(id);
        if (manager == null)
        {
            throw new Exception("Manager not found");
        }
        await _managerRepository.DeleteManagerAsync(id);
    }
    
    public async Task<IEnumerable<ManagerDto>> GetManagersAsync()
    {
        var managers = await _managerRepository.GetManagersAsync();
        return managers.Select(ManagerDto.FromManagerToManagerDto);
    }
    
    public async Task<IEnumerable<AttendanceDto>> GetManagerAttendancesAsync(Guid managerId)
    {
        var manager = await _managerRepository.GetManagerAsync(managerId);
        if (manager == null)
        {
            throw new Exception("Manager not found");
        }
        var attendances = await _attendanceRepository.GetAttendancesByManagerIdAsync(managerId);
        return attendances.Select(AttendanceDto.FromAttendanceToAttendanceDto);
    }
    
    public async Task<AttendanceDto> CreateAttendanceAsync(Guid managerId, AttendanceDto attendanceDto)
    {
        var manager = await _managerRepository.GetManagerAsync(managerId);
        if (manager == null)
        {
            throw new Exception("Manager not found");
        }
        var attendance = Attendance.FromAttendanceDtoToAttendance(attendanceDto);
        attendance.ManagerId = managerId;
        attendance = await _attendanceRepository.CreateAttendanceAsync(attendance);
        return AttendanceDto.FromAttendanceToAttendanceDto(attendance);
    }
    
    public async Task<AttendanceDto> UpdateAttendanceAsync(Guid managerId, Guid attendanceId, AttendanceDto attendanceDto)
    {
        var manager = await _managerRepository.GetManagerAsync(managerId);
        if (manager == null)
        {
            throw new Exception("Manager not found");
        }
        var attendance = await _attendanceRepository.GetAttendanceAsync(attendanceId);
        if (attendance == null)
        {
            throw new Exception("Attendance not found");
        }
        attendance = Attendance.FromAttendanceDtoToAttendance(attendanceDto);
        attendance.Id = attendanceId;
        attendance.ManagerId = managerId;
        attendance = await _attendanceRepository.UpdateAttendanceAsync(attendance);
        return AttendanceDto.FromAttendanceToAttendanceDto(attendance);
    }
    
    
    
    
}