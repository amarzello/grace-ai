"""
Grace AI System - OVOS Commands Module

This module implements command operations for OpenVoiceOS integration.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union

# Import OVOS client for functionality
from .ovos_client import OVOSClient


class OVOSCommands:
    """
    OpenVoiceOS commands implementation.
    
    This class implements specific command operations for interacting with
    OpenVoiceOS functionality, such as controlling volume, brightness,
    and device settings.
    """
    
    def __init__(self, ovos_client: OVOSClient):
        """
        Initialize OVOS commands with the provided client.
        
        Args:
            ovos_client: OVOS client
        """
        self.logger = logging.getLogger('grace.ovos.commands')
        self.client = ovos_client
        
        # Cache for device settings
        self.device_settings = {}
        self.last_settings_update = 0
        self.settings_cache_ttl = 60  # Settings cache TTL in seconds
        
    # Volume control functions
    
    def get_volume(self) -> Optional[int]:
        """
        Get current volume level.
        
        Returns:
            Volume level (0-100) or None if failed
        """
        if not self.client or not self.client.is_connected():
            return None
            
        success, data = self.client.send_and_wait(
            'ovos.volume.get',
            response_type='ovos.volume.get.response',
            timeout=3
        )
        
        if success and data and 'volume' in data:
            return data['volume']
        return None
        
    async def get_volume_async(self) -> Optional[int]:
        """
        Get current volume level asynchronously.
        
        Returns:
            Volume level (0-100) or None if failed
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_volume)
        
    def set_volume(self, level: int) -> bool:
        """
        Set volume level.
        
        Args:
            level: Volume level (0-100)
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        # Ensure level is within valid range
        level = max(0, min(100, level))
        
        return self.client.send_message('ovos.volume.set', {'level': level})
        
    def volume_up(self) -> bool:
        """
        Increase volume.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.volume.increase')
        
    def volume_down(self) -> bool:
        """
        Decrease volume.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.volume.decrease')
        
    def mute(self) -> bool:
        """
        Mute audio.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.volume.mute')
        
    def unmute(self) -> bool:
        """
        Unmute audio.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.volume.unmute')
    
    # Display and brightness functions
    
    def set_brightness(self, level: int) -> bool:
        """
        Set screen brightness.
        
        Args:
            level: Brightness level (0-100)
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        # Ensure level is within valid range
        level = max(0, min(100, level))
        
        return self.client.send_message('brightness', {'level': level})
        
    def get_device_settings(self, force_refresh: bool = False) -> Dict:
        """
        Get device settings.
        
        Args:
            force_refresh: Force refresh of cached settings
            
        Returns:
            Device settings
        """
        # Check if we have a cached version that's still valid
        current_time = time.time()
        if (not force_refresh and 
            self.device_settings and 
            current_time - self.last_settings_update < self.settings_cache_ttl):
            return self.device_settings
            
        if not self.client or not self.client.is_connected():
            return {}
            
        success, data = self.client.send_and_wait(
            'ovos.device.settings',
            timeout=3
        )
        
        if success and data:
            self.device_settings = data
            self.last_settings_update = current_time
            return data
        return {}
    
    async def get_device_settings_async(self, force_refresh: bool = False) -> Dict:
        """
        Get device settings asynchronously.
        
        Args:
            force_refresh: Force refresh of cached settings
            
        Returns:
            Device settings
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_device_settings, force_refresh)
    
    # Speech and utterance functions
    
    def speak(self, text: str) -> bool:
        """
        Make OVOS speak text using its TTS.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('speak', {'utterance': text})
        
    def send_utterance(self, utterance: str) -> bool:
        """
        Send an utterance to be processed by skills.
        
        Args:
            utterance: Text to process
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('recognizer_loop:utterance', 
                                      {'utterances': [utterance]})
    
    # System control functions
    
    def stop(self) -> bool:
        """
        Send stop command to stop all skills.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.stop')
        
    def restart_services(self) -> bool:
        """
        Restart OVOS services.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('system.ovos.service.restart')
        
    def reboot_device(self) -> bool:
        """
        Reboot the device running OVOS.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('system.reboot')
        
    def shutdown_device(self) -> bool:
        """
        Shutdown the device running OVOS.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('system.shutdown')
    
    # Skill management functions
    
    def list_skills(self) -> List[Dict]:
        """
        Get list of installed skills.
        
        Returns:
            List of skill information
        """
        if not self.client or not self.client.is_connected():
            return []
            
        success, data = self.client.send_and_wait(
            'skillmanager.list',
            response_type='skillmanager.list.response',
            timeout=5
        )
        
        if success and data and 'skills' in data:
            return data['skills']
        return []
        
    async def list_skills_async(self) -> List[Dict]:
        """
        Get list of installed skills asynchronously.
        
        Returns:
            List of skill information
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.list_skills)
        
    def activate_skill(self, skill_id: str) -> bool:
        """
        Activate a skill.
        
        Args:
            skill_id: Skill identifier
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        success, data = self.client.send_and_wait(
            'skillmanager.activate',
            {'skill_id': skill_id},
            response_type='skillmanager.activate.response',
            timeout=5
        )
        
        return success and data and data.get('success', False)
        
    def deactivate_skill(self, skill_id: str) -> bool:
        """
        Deactivate a skill.
        
        Args:
            skill_id: Skill identifier
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        success, data = self.client.send_and_wait(
            'skillmanager.deactivate',
            {'skill_id': skill_id},
            response_type='skillmanager.deactivate.response',
            timeout=5
        )
        
        return success and data and data.get('success', False)
        
    def install_skill(self, skill_url: str) -> bool:
        """
        Install a skill.
        
        Args:
            skill_url: URL to the skill repository
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.skills.install', {'url': skill_url})
        
    def uninstall_skill(self, skill_id: str) -> bool:
        """
        Uninstall a skill.
        
        Args:
            skill_id: Skill identifier
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.skills.uninstall', {'skill_id': skill_id})
    
    # Audio player functions
    
    def play_audio(self, uri: str) -> bool:
        """
        Play audio file.
        
        Args:
            uri: URI of audio file
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.play', {'uri': uri})
        
    def pause_audio(self) -> bool:
        """
        Pause audio playback.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.pause')
        
    def resume_audio(self) -> bool:
        """
        Resume audio playback.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.resume')
        
    def stop_audio(self) -> bool:
        """
        Stop audio playback.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.stop')
        
    def next_track(self) -> bool:
        """
        Skip to next track.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.next')
        
    def prev_track(self) -> bool:
        """
        Go to previous track.
        
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message('ovos.audio.service.prev')
        
    # HomeAssistant integration functions
    
    def ha_turn_on_device(self, device_id: str) -> bool:
        """
        Turn on a HomeAssistant device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message(
            'ovos.phal.plugin.homeassistant.device.turn_on',
            {'device_id': device_id}
        )
        
    def ha_turn_off_device(self, device_id: str) -> bool:
        """
        Turn off a HomeAssistant device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.client.send_message(
            'ovos.phal.plugin.homeassistant.device.turn_off',
            {'device_id': device_id}
        )
        
    def ha_get_devices(self) -> List[Dict]:
        """
        Get list of HomeAssistant devices.
        
        Returns:
            List of device information
        """
        if not self.client or not self.client.is_connected():
            return []
            
        success, data = self.client.send_and_wait(
            'ovos.phal.plugin.homeassistant.get.devices',
            response_type='ovos.phal.plugin.homeassistant.get.devices.response',
            timeout=5
        )
        
        if success and data and 'devices' in data:
            return data['devices']
        return []
