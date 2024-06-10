using Microsoft.EntityFrameworkCore;
using Models;

namespace Services;

    public class ManagerRepository: IManagerRepository
    {
        private readonly MyDbContext _context;
        private readonly DbSet<Manager> _managers;

        public ManagerRepository(MyDbContext context)
        {
            _context = context;
            _managers = context.Managers;
        
        }
        
        public async Task<IEnumerable<Manager>> GetManagersAsync()
        {
            return await _managers.ToListAsync();
        }
        
        public async Task<Manager> GetManagerAsync(Guid id)
        {
            return await _managers.FindAsync(id);
        }
        
        public async Task<Manager> GetManagerByEmailAsync(string email)
        {
            return await _managers.FirstOrDefaultAsync(m => m.Email == email);
        }
        
        public async Task<Manager> CreateManagerAsync(Manager manager)
        {
            var entry = await _managers.AddAsync(manager);
            await _context.SaveChangesAsync();
            return entry.Entity;
        }
        
        public async Task<Manager> UpdateManagerAsync(Manager manager)
        {
            var entry = _managers.Update(manager);
            await _context.SaveChangesAsync();
            return entry.Entity;
        }
        
        public async Task<Manager> DeleteManagerAsync(Guid id)
        {
            var manager = await _managers.FindAsync(id);
            if (manager == null)
            {
                return null;
            }
            _managers.Remove(manager);
            await _context.SaveChangesAsync();
            return manager;
        }
        
        public async Task<IEnumerable<Attendance>> GetAttendancesAsync(Guid id)
        {
            return await _context.Attendances.Where(a => a.ManagerId == id).ToListAsync();
        }
    }
