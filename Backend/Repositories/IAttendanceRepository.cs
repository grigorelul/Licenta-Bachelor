using Models;

namespace Repositories;

public interface IAttendanceRepository
{
    Task<IEnumerable<Attendance>> GetAttendancesAsync();
    Task<Attendance> GetAttendanceAsync(Guid id);
    Task<Attendance> CreateAttendanceAsync(Attendance attendance);
    Task<Attendance> UpdateAttendanceAsync(Attendance attendance);
    Task<Attendance> DeleteAttendanceAsync(Guid id);
    Task<IEnumerable<Attendance>> GetAttendancesByUserIdAsync(Guid userId);
    Task<IEnumerable<Attendance>> GetAttendancesByManagerIdAsync(Guid managerId);
    
}