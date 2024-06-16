using Models;
using Repositories;
namespace Services;

public interface IManagerService
{
    Task<ManagerDto> GetManagerAsync(Guid id);
    Task<ManagerDto> CreateManagerAsync(ManagerDto managerDto);
    Task<ManagerDto> UpdateManagerAsync(Guid id, ManagerDto managerDto);
    Task DeleteManagerAsync(Guid id);
    Task<IEnumerable<ManagerDto>> GetManagersAsync();
    Task<IEnumerable<AttendanceDto>> GetManagerAttendancesAsync(Guid managerId);
    Task<AttendanceDto> CreateAttendanceAsync(Guid managerId, AttendanceDto attendanceDto);
    Task<AttendanceDto> UpdateAttendanceAsync(Guid managerId, Guid attendanceId, AttendanceDto attendanceDto);
    
}