using Repositories;
using Models;

namespace Services;

public class AttendanceService : IAttendanceService
{
    
    private readonly IAttendanceRepository _attendanceRepository;
    private readonly IUserRepository _userRepository;
    private readonly IManagerRepository _managerRepository;

    public AttendanceService(IAttendanceRepository attendanceRepository, IUserRepository userRepository, IManagerRepository managerRepository)
    {
        _attendanceRepository = attendanceRepository;
        _userRepository = userRepository;
        _managerRepository = managerRepository;
    }

    public async Task<IEnumerable<Attendance>> GetAttendancesAsync()
    {
        return await _attendanceRepository.GetAttendancesAsync();
    }

    public async Task<Attendance> GetAttendanceAsync(Guid id)
    {
        return await _attendanceRepository.GetAttendanceAsync(id);
    }

    public async Task<Attendance> CreateAttendanceAsync(Attendance attendance)
    {
        var user = await _userRepository.GetUserAsync(attendance.UserId);
        var manager = await _managerRepository.GetManagerAsync(attendance.ManagerId);
        if (user == null || manager == null)
        {
            throw new Exception("User or Manager not found");
        }
        return await _attendanceRepository.CreateAttendanceAsync(attendance);
    }

    public async Task<Attendance> UpdateAttendanceAsync(Attendance attendance)
    {
        var user = await _userRepository.GetUserAsync(attendance.UserId);
        var manager = await _managerRepository.GetManagerAsync(attendance.ManagerId);
        if (user == null || manager == null)
        {
            throw new Exception("User or Manager not found");
        }
        return await _attendanceRepository.UpdateAttendanceAsync(attendance);
    }

    public async Task<Attendance> DeleteAttendanceAsync(Guid id)
    {
        return await _attendanceRepository.DeleteAttendanceAsync(id);
    }

    public async Task<IEnumerable<Attendance>> GetAttendancesByUserIdAsync(Guid userId)
    {
        return await _attendanceRepository.GetAttendancesByUserIdAsync(userId);
    }

    public async Task<IEnumerable<Attendance>> GetAttendancesByManagerIdAsync(Guid managerId)
    {
        return await _attendanceRepository.GetAttendancesByManagerIdAsync(managerId);
    }
}