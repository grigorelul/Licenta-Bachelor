using Microsoft.EntityFrameworkCore;
using Models;

namespace Services;

public class AttendanceRepository : IAttendanceRepository
{
    private readonly MyDbContext _context;
    private readonly DbSet<Attendance> _attendances;

    public AttendanceRepository(MyDbContext context)
    {
        _context = context;
        _attendances = context.Attendances;
    }
    
    public async Task<IEnumerable<Attendance>> GetAttendancesAsync()
    {
        return await _attendances.ToListAsync();
    }
    
    public async Task<Attendance> GetAttendanceAsync(Guid id)
    {
        return await _attendances.FindAsync(id);
    }
    
    public async Task<Attendance> CreateAttendanceAsync(Attendance attendance)
    {
        var entry = await _attendances.AddAsync(attendance);
        await _context.SaveChangesAsync();
        return entry.Entity;
    }
    
    public async Task<Attendance> UpdateAttendanceAsync(Attendance attendance)
    {
        var entry = _attendances.Update(attendance);
        await _context.SaveChangesAsync();
        return entry.Entity;
    }
   
    public async Task<Attendance> DeleteAttendanceAsync(Guid id)
    {
        var attendance = await _attendances.FindAsync(id);
        if (attendance == null)
        {
            return null;
        }
        _attendances.Remove(attendance);
        await _context.SaveChangesAsync();
        return attendance;
    }
    
    public async Task<IEnumerable<Attendance>> GetAttendancesByUserIdAsync(Guid userId)
    {
        return await _attendances.Where(a => a.UserId == userId).ToListAsync();
    }
    
    public async Task<IEnumerable<Attendance>> GetAttendancesByManagerIdAsync(Guid managerId)
    {
        return await _attendances.Where(a => a.ManagerId == managerId).ToListAsync();
    }
    
}