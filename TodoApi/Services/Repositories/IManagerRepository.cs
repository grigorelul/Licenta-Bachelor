using TodoApi.Models;

namespace Services
{
    public interface IManagerRepository
    {
        Task<IEnumerable<Manager>> GetManagersAsync();
        Task<Manager> GetManagerAsync(Guid id);
        Task<Manager> GetManagerByEmailAsync(string email);
        Task<Manager> CreateManagerAsync(Manager manager);
        Task<Manager> UpdateManagerAsync(Manager manager);
        Task<Manager> DeleteManagerAsync(Guid id);
        Task<IEnumerable<Attendance>> GetAttendancesAsync(Guid id);
    }
}
