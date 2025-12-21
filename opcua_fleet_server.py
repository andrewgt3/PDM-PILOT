import asyncio
import logging
import random
from asyncua import Server, ua

# --- CONFIGURATION ---
ROBOT_IDS = ["ROBOT_1", "ROBOT_2", "ROBOT_3", "ROBOT_4"]
CYCLE_DURATION_HEALTHY = 45  # Seconds of Green
CYCLE_DURATION_FAILURE = 30  # Seconds of Red

async def chaos_monkey(cascade_trigger_node):
    """
    This background task automatically toggles the failure mode.
    """
    while True:
        # PHASE 1: NORMAL OPERATION
        print(f"✅ SYSTEM HEALTHY. Next failure in {CYCLE_DURATION_HEALTHY}s...")
        await cascade_trigger_node.write_value(False)
        await asyncio.sleep(CYCLE_DURATION_HEALTHY)
        
        # PHASE 2: CASCADE FAILURE
        print(f"🚨 TRIGGERING CASCADE FAILURE! Duration: {CYCLE_DURATION_FAILURE}s...")
        await cascade_trigger_node.write_value(True)
        await asyncio.sleep(CYCLE_DURATION_FAILURE)

async def main():
    # Setup Server
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")
    server.set_server_name("Gaia Virtual Fleet")

    # Setup Namespace
    idx = await server.register_namespace("http://gaiapredictive.com/fleet")
    objects = server.nodes.objects

    # --- BUILD THE FLEET ---
    robot_nodes = {}
    
    # Create the Trigger Variable
    cascade_trigger = await objects.add_variable(idx, "Cascade_Failure_Active", False)
    await cascade_trigger.set_writable()
    
    print(f"🏗️ Building Virtual Fleet ({len(ROBOT_IDS)} Assets)...")
    
    for r_id in ROBOT_IDS:
        obj = await objects.add_object(idx, r_id)
        
        # Add Sensors
        vib = await obj.add_variable(idx, "Vibration_X", 0.1)
        trq = await obj.add_variable(idx, "Torque_J1", 45.0)
        tmp = await obj.add_variable(idx, "Motor_Temp", 55.0)
        
        await vib.set_writable()
        await trq.set_writable()
        await tmp.set_writable()
        
        robot_nodes[r_id] = {'vib': vib, 'trq': trq, 'tmp': tmp}

    print("✅ Server Started at opc.tcp://localhost:4840")
    
    # START THE CHAOS MONKEY (The Automation)
    asyncio.create_task(chaos_monkey(cascade_trigger))
    
    async with server:
        while True:
            # Check Failure Status (set by the Chaos Monkey)
            is_cascade = await cascade_trigger.read_value()
            
            for r_id, sensors in robot_nodes.items():
                v_noise = random.uniform(-0.01, 0.01)
                
                if not is_cascade:
                    # NORMAL (NASA Physics: Nominal)
                    new_vib = 0.12 + v_noise
                    new_trq = 45.0 + random.uniform(-2, 2)
                    new_tmp = 55.0 + random.uniform(-0.5, 0.5)
                
                else:
                    # FAILURE (NASA Physics: Critical Fault)
                    if r_id == "ROBOT_1": # The Source
                        new_vib = 2.5 + random.uniform(-0.2, 0.2) 
                        new_trq = 180.0 + random.uniform(-10, 10) 
                        new_tmp = 85.0 + random.uniform(0, 0.5)   
                    elif r_id in ["ROBOT_2", "ROBOT_3"]: # The Victims
                        new_vib = 0.01 + (v_noise * 0.1)          
                        new_trq = 0.0                             
                        new_tmp = 55.0 - 0.1                      
                    elif r_id == "ROBOT_4": # The Compensator
                        new_vib = 0.65 + v_noise                  
                        new_trq = 50.0 + random.uniform(-5, 5)
                        new_tmp = 65.0 + random.uniform(0, 0.2)

                await sensors['vib'].write_value(new_vib)
                await sensors['trq'].write_value(new_trq)
                await sensors['tmp'].write_value(new_tmp)

            await asyncio.sleep(0.5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
